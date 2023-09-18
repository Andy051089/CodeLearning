/**
 * 輸入年分，此年為閏年還是平年
 */
package hello;
import java.util.Scanner;
public class IfElseDate {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入一個年份");
		int y = input.nextInt();
		if((y/4==0 && y/100!=0) || y/400==0) {
			System.out.println("此為閏年");
		}else {
			System.out.println("此為平年");
		}
	}
}
