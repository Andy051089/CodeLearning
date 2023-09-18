/**
 * 輸入月分告訴你有機天
 */
package hello;
import java.util.Scanner;
public class HowManyDays {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入月份");
		int month = input.nextInt();
		switch(month) {
		case 1 :
			System.out.println("這個月有31天");
			break;
		case 2 :
			System.out.println("今年是哪個年份");
			int year = input.nextInt();
			if((year/4==0 && year/100!=0) || year/400==0) {
				System.out.println("這個月有29天");
			}else {
				System.out.println("這個月有28天");
			}
			break;	
		case 3 :
			System.out.println("這個月有31天");
			break;
		case 4 :
			System.out.println("這個月有30天");
			break;
		case 5 :
			System.out.println("這個月有31天");
			break;
		case 6 :
			System.out.println("這個月有30天");
			break;
		case 7 :
			System.out.println("這個月有31天");
			break;
		case 8 :
			System.out.println("這個月有31天");
			break;
		case 9 :
			System.out.println("這個月有30天");
			break;
		case 10 :
			System.out.println("這個月有31天");
			break;
		case 11 :
			System.out.println("這個月有30天");
			break;
		case 12 :
			System.out.println("這個月有31天");
			break;
		default:
			System.out.println("沒有這個月份");
			break;
		}
	}
}
