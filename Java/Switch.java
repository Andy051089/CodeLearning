/**
 * 輸入1-7的數字對應星期一至星期日
 */
package hello;
import java.util.Scanner;
public class Switch {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入1-7的數字");
		int day = input.nextInt();
		switch(day) {
		case 1:
			System.out.println("星期一");
			break;
		case 2:
			System.out.println("星期二");
			break;
		case 3:
			System.out.println("星期三");
			break;
		case 4:
			System.out.println("星期四");
			break;
		case 5:
			System.out.println("星期五");
			break;
		case 6:
			System.out.println("星期六");
			break;
		case 7:
			System.out.println("星期日");
			break;
		default:
			System.out.println("輸入超過7的數字");
			break;
		}
		
	}
}
